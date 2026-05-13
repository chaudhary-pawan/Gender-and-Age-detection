#!/bin/bash
set -o errexit

pip install -r requirements.txt
python manage.py collectstatic --no-input

if [ "${RUN_MIGRATIONS}" = "True" ] || [ "${RUN_MIGRATIONS}" = "true" ]; then
  python manage.py migrate --no-input
fi
