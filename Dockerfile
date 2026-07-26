# Base image. slim = smaller. 3.11 MUST match — your pyfunc was pickled under 3.11.
FROM python:3.11-slim

# All paths inside the container are relative to /app.
WORKDIR /app

# Copy ONLY requirements first. Docker caches layers: deps reinstall only when
# this file changes, not on every code edit. Big speed win on rebuilds.
COPY requirements.txt .

# Install everything. --no-cache-dir trims image size. en_core_web_sm is already
# a wheel URL in requirements.txt, so spaCy's model lands here too — no extra step.
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake NLTK corpora so startup is fast and doesn't depend on NLTK's servers.
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet

# Point your load_context (os.environ.get("NLTK_DATA")) at that baked data.
ENV NLTK_DATA=/usr/share/nltk_data

# Copy app code LAST (changes most often → keeps the slow deps layer cached).
COPY flask_app/ ./flask_app/

# Document the port the API serves on.
EXPOSE 5000

# Production server (waitress is already in requirements). host=0.0.0.0 makes it
# reachable from outside the container.
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "flask_app.app:app"]