# get the base node image
FROM node:lts-slim as builder

# set the working dir for container
WORKDIR /frontend

# copy the json file first
COPY ./package.json /frontend

# install npm dependencies
RUN npm install yarn -g --force && yarn install

# copy other project files
COPY . /frontend

# build the folder
RUN yarn run build && yarn global add serve

# Handle Nginx
FROM nginxinc/nginx-unprivileged
COPY --from=builder /frontend/build /usr/share/nginx/html
COPY ./nginx/default.conf /etc/nginx/conf.d/default.conf
