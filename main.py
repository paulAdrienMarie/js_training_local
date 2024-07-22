from aiohttp import web
from routes import setup_routes

app = web.Application(client_max_size=1024**2 * 100)
setup_routes(app)

web.run_app(app)

