from fastapi import Request


def get_current_username(request: Request) -> str | None:
    return request.session.get("username")


def require_user(request: Request) -> str | None:
    return get_current_username(request)
