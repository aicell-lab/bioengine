from typing import Any, Dict, Union, List, Optional


def create_context(
    user_id: Optional[str] = None, user_email: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Create a context dictionary for user authentication and authorization.

    Creates a standardized context structure containing user information that
    can be used throughout the BioEngine worker system for permission checking
    and audit logging.

    Args:
        user_id: User identifier, defaults to "anonymous" if not provided
        user_email: User email address, defaults to "anonymous@example.com" if not provided

    Returns:
        Dict: Context dictionary with standardized user information structure:
              {"user": {"id": str, "email": str}}
    """
    return {
        "user": {
            "id": user_id if user_id else "anonymous",
            "email": user_email if user_email else "anonymous@example.com",
        }
    }


def check_permissions(
    context: Optional[Dict[str, Any]],
    authorized_users: Union[List[str], str, None],
    resource_name: str,
    allow_wildcard: bool = True,
) -> None:
    """
    Check if the user in the context is authorized to access a resource.

    Validates user permissions against the authorized users list for specific
    resource operations. Supports both individual users and wildcard access.
    Provides comprehensive error messages for debugging and audit purposes.

    Args:
        context: Request context containing user information with structure:
                {"user": {"id": str, "email": str}}
        authorized_users: List of authorized user IDs/emails, single user string,
                         or None. Special values:
                         - ["*"] or "*": Allows all users (wildcard access)
                         - None or []: Denies all access
        resource_name: Name of the resource being accessed for error messaging
        allow_wildcard: Whether a "*" entry authorizes every caller. Pass False for
                        operations that widen the authorized set itself — a caller
                        who is only covered by "*" must not be able to make that
                        grant permanent or revoke it for everyone else.

    Raises:
        PermissionError: If user is not authorized to access the resource with
                        detailed error message explaining the specific failure
    """
    # Validate context structure
    if context is None or not isinstance(context, dict) or "user" not in context:
        raise PermissionError(
            f"Invalid context for {resource_name}: missing user information. "
            "Expected context with 'user' field containing user details."
        )

    user = context["user"]
    if not isinstance(user, dict):
        raise PermissionError(
            f"Invalid user information in context for {resource_name}: "
            "user field must be a dictionary."
        )

    # Extract user identifiers
    user_id = user.get("id")
    user_email = user.get("email")

    if not user_id and not user_email:
        raise PermissionError(
            f"Invalid user context for {resource_name}: "
            "user must have either 'id' or 'email' field."
        )

    # Handle None or empty authorized_users (deny all access)
    if authorized_users is None or authorized_users == []:
        raise PermissionError(
            f"Access denied for {resource_name}: no users are authorized to access this resource."
        )

    # Convert single string to list for uniform processing
    if isinstance(authorized_users, str):
        authorized_users = [authorized_users]

    # Check for wildcard access
    if "*" in authorized_users:
        if allow_wildcard:
            return  # Wildcard access - all users allowed
        # Fall through on the named entries only, so the denial reports them.
        authorized_users = [user for user in authorized_users if user != "*"]

    # Check specific user authorization
    if user_id and user_id in authorized_users:
        return  # User ID match

    if user_email and user_email in authorized_users:
        return  # Email match

    # Access denied - provide detailed error message
    raise PermissionError(
        f"User '{user_id or 'no id'}' ({user_email or 'no-email'}) is not authorized for {resource_name}. "
        f"Authorized users: {authorized_users}"
    )
