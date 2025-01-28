FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# Build argument for API URL
ARG REACT_APP_API_URL

# Set environment variable for build
ENV REACT_APP_API_URL $REACT_APP_API_URL

# Build the React app
RUN npm run build

# Install 'serve' to serve static files
RUN npm install -g serve

EXPOSE 5000

CMD ["serve", "-s", "build", "-l", "5000"]