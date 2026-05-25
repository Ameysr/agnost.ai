FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# VITE_API_URL is injected at build time so the compiled JS bundle
# knows where to reach the backend container
ARG VITE_API_URL=http://localhost:8050
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# Serve the static build with a lightweight nginx image
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

# Simple nginx config that handles SPA client-side routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
