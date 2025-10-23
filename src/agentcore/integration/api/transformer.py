"""Response transformation and validation.

Transforms and validates API responses with support for JSON, pagination, and custom transformations.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

from agentcore.integration.api.exceptions import APITransformationError, APIValidationError
from agentcore.integration.api.models import ResponseTransformation

logger = structlog.get_logger(__name__)


class ResponseTransformer:
    """Transform and validate API responses.

    Supports:
    - JSON parsing and validation
    - Pydantic model validation
    - Custom transformation pipelines
    - Error response handling
    - Pagination support
    """

    def __init__(self, transformation: ResponseTransformation | None = None) -> None:
        """Initialize response transformer.

        Args:
            transformation: Optional transformation configuration
        """
        self.transformation = transformation

    def transform(
        self,
        raw_body: str | bytes,
        content_type: str | None = None,
    ) -> Any:
        """Transform raw response body to structured data.

        Args:
            raw_body: Raw response body
            content_type: Content-Type header value

        Returns:
            Transformed response data

        Raises:
            APITransformationError: If transformation fails
        """
        # Parse based on content type
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")

        parsed = self._parse_response(raw_body, content_type)

        # Apply transformation rules if configured
        if self.transformation and self.transformation.rules:
            try:
                parsed = self._apply_transformation_rules(parsed)
            except Exception as e:
                if self.transformation.error_handling == "raise":
                    raise APITransformationError(f"Transformation failed: {e}") from e
                if self.transformation.error_handling == "default":
                    return self.transformation.default_value
                # Skip transformation on error
                logger.warning("transformation_failed", error=str(e))

        return parsed

    def _parse_response(self, raw_body: str, content_type: str | None) -> Any:
        """Parse response based on content type.

        Args:
            raw_body: Raw response body
            content_type: Content-Type header value

        Returns:
            Parsed response data

        Raises:
            APITransformationError: If parsing fails
        """
        if not raw_body:
            return None

        # Determine parser from content type
        if content_type:
            content_type_lower = content_type.lower()

            if "application/json" in content_type_lower:
                return self._parse_json(raw_body)
            if "text/plain" in content_type_lower:
                return raw_body
            if "text/html" in content_type_lower:
                return raw_body
            if "application/xml" in content_type_lower or "text/xml" in content_type_lower:
                return self._parse_xml(raw_body)

        # Try JSON first (most common)
        try:
            return self._parse_json(raw_body)
        except APITransformationError:
            # Return as-is if not JSON
            return raw_body

    def _parse_json(self, raw_body: str) -> Any:
        """Parse JSON response.

        Args:
            raw_body: Raw JSON string

        Returns:
            Parsed JSON data

        Raises:
            APITransformationError: If JSON parsing fails
        """
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise APITransformationError(f"Failed to parse JSON response: {e}") from e

    def _parse_xml(self, raw_body: str) -> Any:
        """Parse XML response (basic implementation).

        Args:
            raw_body: Raw XML string

        Returns:
            Parsed XML data as dict (basic implementation returns raw string)

        Note:
            Full XML parsing would require xml.etree.ElementTree or lxml
        """
        # Basic implementation - return raw XML
        # For production use, implement proper XML parsing
        logger.warning("xml_parsing_not_implemented")
        return raw_body

    def _apply_transformation_rules(self, data: Any) -> Any:
        """Apply transformation rules to parsed data.

        Args:
            data: Parsed response data

        Returns:
            Transformed data

        Raises:
            APITransformationError: If transformation fails
        """
        if not self.transformation or not self.transformation.rules:
            return data

        result: dict[str, Any] = {}

        for rule in self.transformation.rules:
            if rule.rule_type == "extract":
                extracted = self._extract_value(data, rule.source_path)
                if rule.target_path:
                    self._set_value(result, rule.target_path, extracted)
                else:
                    return extracted

            elif rule.rule_type == "map":
                # Map transformation (not fully implemented)
                logger.warning("map_transformation_not_implemented")

            elif rule.rule_type == "filter":
                # Filter transformation (not fully implemented)
                logger.warning("filter_transformation_not_implemented")

            elif rule.rule_type == "validate":
                # Validation (not fully implemented)
                logger.warning("validate_transformation_not_implemented")

        return result if result else data

    def _extract_value(self, data: Any, path: str) -> Any:
        """Extract value from data using dot notation path.

        Args:
            data: Source data
            path: Dot notation path (e.g., "data.items[0].name")

        Returns:
            Extracted value

        Raises:
            APITransformationError: If extraction fails
        """
        if not path:
            return data

        parts = path.split(".")
        current = data

        for part in parts:
            # Handle array indexing
            if "[" in part and "]" in part:
                field = part[: part.index("[")]
                index_str = part[part.index("[") + 1 : part.index("]")]

                if field:
                    current = current.get(field) if isinstance(current, dict) else getattr(current, field, None)

                try:
                    index = int(index_str)
                    current = current[index] if current else None
                except (ValueError, IndexError, TypeError) as e:
                    raise APITransformationError(
                        f"Failed to extract array index {index_str} from path {path}"
                    ) from e
            else:
                # Regular field access
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    current = getattr(current, part, None)

            if current is None:
                return None

        return current

    def _set_value(self, target: dict[str, Any], path: str, value: Any) -> None:
        """Set value in target dict using dot notation path.

        Args:
            target: Target dictionary
            path: Dot notation path
            value: Value to set
        """
        parts = path.split(".")
        current = target

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def validate(self, data: Any, model: type[BaseModel]) -> BaseModel:
        """Validate response data against Pydantic model.

        Args:
            data: Response data to validate
            model: Pydantic model class

        Returns:
            Validated model instance

        Raises:
            APIValidationError: If validation fails
        """
        try:
            return model.model_validate(data)
        except ValidationError as e:
            raise APIValidationError(f"Response validation failed: {e}") from e

    def extract_pagination(self, data: Any) -> dict[str, Any] | None:
        """Extract pagination information from response.

        Supports common pagination patterns:
        - Cursor-based: {data: [...], next_cursor: "...", has_more: true}
        - Offset-based: {data: [...], total: 100, offset: 0, limit: 20}
        - Page-based: {data: [...], page: 1, total_pages: 5}

        Args:
            data: Response data

        Returns:
            Pagination info or None if not paginated
        """
        if not isinstance(data, dict):
            return None

        pagination: dict[str, Any] = {}

        # Check for cursor-based pagination
        if "next_cursor" in data or "cursor" in data:
            pagination["type"] = "cursor"
            pagination["next_cursor"] = data.get("next_cursor") or data.get("cursor")
            pagination["has_more"] = data.get("has_more", True)

        # Check for offset-based pagination
        elif "offset" in data or "limit" in data:
            pagination["type"] = "offset"
            pagination["offset"] = data.get("offset", 0)
            pagination["limit"] = data.get("limit", 20)
            pagination["total"] = data.get("total")

        # Check for page-based pagination
        elif "page" in data or "total_pages" in data:
            pagination["type"] = "page"
            pagination["page"] = data.get("page", 1)
            pagination["total_pages"] = data.get("total_pages")
            pagination["per_page"] = data.get("per_page", 20)

        return pagination if pagination else None

    def extract_errors(self, data: Any) -> list[dict[str, Any]]:
        """Extract error information from response.

        Supports common error formats:
        - {error: "message"}
        - {errors: [{field: "name", message: "required"}]}
        - {message: "error message", code: "ERROR_CODE"}

        Args:
            data: Response data

        Returns:
            List of error dictionaries
        """
        if not isinstance(data, dict):
            return []

        errors: list[dict[str, Any]] = []

        # Single error message
        if "error" in data:
            error_msg = data["error"]
            if isinstance(error_msg, str):
                errors.append({"message": error_msg})
            elif isinstance(error_msg, dict):
                errors.append(error_msg)

        # Multiple errors
        if "errors" in data:
            error_list = data["errors"]
            if isinstance(error_list, list):
                for err in error_list:
                    if isinstance(err, str):
                        errors.append({"message": err})
                    elif isinstance(err, dict):
                        errors.append(err)

        # Error with code
        if "message" in data and "code" in data:
            errors.append({
                "message": data["message"],
                "code": data["code"],
            })

        return errors


# Global transformer instance
_default_transformer: ResponseTransformer | None = None


def get_default_transformer() -> ResponseTransformer:
    """Get the default response transformer instance.

    Returns:
        Default ResponseTransformer instance
    """
    global _default_transformer
    if _default_transformer is None:
        _default_transformer = ResponseTransformer()
    return _default_transformer
