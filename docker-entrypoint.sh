#!/bin/sh
set -e

# Load environment variables from .env file

if [ -f /app/.env ]; then
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip empty lines and comments
        [ -z "$key" ] && continue
        case "$key" in \#*) continue ;; esac
        case "$key" in *[[:space:]]*) continue ;; esac

        # Check if variable is already set in environment
        eval "is_set=\"\${$key+is_set}\""
        if [ "$is_set" != "is_set" ]; then
            export "$key=$value"
        fi
    done < /app/.env
fi

exec "$@"
