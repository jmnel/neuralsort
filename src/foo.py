import settings
import quandl
from pprint import pprint

a = quandl.get_table('ZACKS/FC', ticker='AAPL',
                     api_key=settings.QUANDL_API_KEY)

pprint(a)
