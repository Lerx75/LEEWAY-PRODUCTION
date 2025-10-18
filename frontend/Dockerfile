FROM node:20-bullseye-slim AS deps
WORKDIR /web
COPY package.json package-lock.json* .npmrc* ./
RUN npm ci || npm install

FROM node:20-bullseye-slim AS builder
WORKDIR /web
COPY --from=deps /web/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Minimal runtime using Next.js standalone output
FROM node:20-bullseye-slim AS runner
WORKDIR /web
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000
# Copy standalone server and assets
COPY --from=builder /web/.next/standalone ./
COPY --from=builder /web/.next/static ./.next/static
COPY --from=builder /web/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
