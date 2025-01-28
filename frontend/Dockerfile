FROM node:16 AS build

WORKDIR /app

COPY requirements.txt /app/
COPY frontend/package*.json /app/

RUN npm install

COPY frontend/ /app

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]