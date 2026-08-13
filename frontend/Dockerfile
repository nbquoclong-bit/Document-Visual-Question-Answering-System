# ==============================================================================
# Document Visual Question Answering (DocVQA) System - Frontend Dockerfile
# ==============================================================================

# --- Stage 1: Build Stage ---
FROM node:20-alpine AS build

WORKDIR /app

# Copy package descriptors for optimal layer caching
COPY package.json package-lock.json* ./

# Install dependencies deterministically
RUN npm ci || npm install

# Copy source code and configuration files
COPY . ./

# Build arguments for dynamic environment configurations
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Build application for production (Vite outputs to /app/dist)
RUN npm run build

# --- Stage 2: Production Web Server ---
FROM nginx:1.25-alpine AS production

# Install wget for container healthcheck
RUN apk add --no-cache wget

# Remove default nginx static assets and config
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration for SPA routing & API reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static bundle from build stage (Vite output directory: dist)
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]