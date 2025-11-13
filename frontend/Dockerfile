# build stage
FROM node:18-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# production stage: serve static files with nginx
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
# optional: provide a basic nginx config if needed; default is fine
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
