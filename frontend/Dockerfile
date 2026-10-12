# Stage 1: build the React app with Vite
FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: serve the static files with Nginx and proxy /api/predict
FROM nginx:1.28-alpine

# Files in /etc/nginx/templates are rendered into /etc/nginx/conf.d at start.
# NGINX_ENTRYPOINT_LOCAL_RESOLVERS (set in compose.yaml) fills in the
# container DNS resolver, so Nginx finds the backend again after it restarts.
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
