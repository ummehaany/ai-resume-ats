"""Shared error display for calls to the backend API."""
import requests
import streamlit as st


def show_backend_error(exc: Exception) -> None:
    """Translate a `requests` exception into a friendly Streamlit message."""
    if isinstance(exc, requests.ConnectionError):
        st.error("Could not reach the backend service. Please try again in a moment.")
    elif isinstance(exc, requests.Timeout):
        st.error("The backend took too long to respond. Try again, or use a smaller resume.")
    elif isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        if status == 401:
            # Access tokens expire (Supabase default: 1 hour) - make the user sign in again.
            for key in ("access_token", "refresh_token", "user_id", "user_email"):
                st.session_state[key] = None
            st.warning("Your session has expired. Please sign in again from the sidebar.")
        elif status == 429:
            st.warning(f"Slow down a little: {detail}")
        else:
            st.error(f"Backend returned {status}: {detail}")
    else:
        st.error(f"Unexpected error: {exc}")
