import os

api_key = os.environ.get("AUTH_KEY")

def api_key_auth(func):

    def wrapper(handler, *args, **kwargs):
        if api_key:
            req_api_key = handler.request.headers.get("x-api-key")
            if req_api_key != api_key:
                handler.set_status(401)
                handler.write({"message": "unauthorized"})
                return
        func(handler, *args, **kwargs)

    return wrapper