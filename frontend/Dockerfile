# Build the console, then serve it from nginx, which also proxies /api to the api service.
FROM node:20-alpine AS build
WORKDIR /ui
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# src/api/schema.d.ts is committed, so this build needs neither Python nor a running API.
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/dist /usr/share/nginx/html
EXPOSE 80
