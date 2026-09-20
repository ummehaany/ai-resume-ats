"""Google OAuth (PKCE) mechanics, checked against the REAL supabase-py library with a made-up project URL.

No network, no Google, no Supabase project involved: this only proves (a) the private supabase-py attributes
that frontend/services/supabase_client.py relies on still exist in the pinned version, and (b) the documented
limitation - a second sign-in started on the shared client overwrites the first one's PKCE verifier.
The actual Google round trip can only be tested by hand (see README).
"""
import urllib.parse as up

from supabase import create_client

URL, KEY = 'https://abcdefghijklmnopqrst.supabase.co', 'anon-key-not-real-aaaaaaaaaaaaaaaaaaaaaaaaaaaa'


def _start(client):
    return client.auth.sign_in_with_oauth({'provider': 'google', 'options': {'redirect_to': 'http://localhost:8501'}})


def test_private_attributes_used_by_the_frontend_still_exist_and_pkce_is_used():
    c = create_client(URL, KEY)
    key = f'{c.auth._storage_key}-code-verifier'
    resp = _start(c)
    query = up.parse_qs(up.urlparse(resp.url).query)
    assert query.get('provider') == ['google'] and 'code_challenge' in query      # PKCE, not the implicit flow
    assert c.auth._storage.get_item(key)                                          # verifier kept in the client's storage


def test_known_limitation_second_signin_overwrites_first_verifier():
    c = create_client(URL, KEY)
    key = f'{c.auth._storage_key}-code-verifier'
    _start(c)
    first = c.auth._storage.get_item(key)
    _start(c)
    assert c.auth._storage.get_item(key) != first        # user A's exchange would now fail; A just clicks "Sign in" again
