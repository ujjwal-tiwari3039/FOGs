#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting the ED-05 Full-Stack Application...${NC}\n"

# 1. Start the FastAPI backend in the background
echo -e "${GREEN}==> Booting Python FastAPI Backend (Port 8000)...${NC}"
python3 api.py &
BACKEND_PID=$!

# Give the backend a second to initialize
sleep 2

# 2. Start the Next.js frontend
echo -e "\n${GREEN}==> Booting Next.js React Frontend (Port 3000)...${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!

# 3. Clean shutdown handling
# When you press Ctrl+C, this ensures both the frontend and backend servers are killed.
cleanup() {
    echo -e "\n${BLUE}Shutting down servers...${NC}"
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

# Catch termination signals
trap cleanup SIGINT SIGTERM

echo -e "\n${BLUE}App is running! Press Ctrl+C to stop both servers.${NC}"
echo -e "Access the UI at: ${GREEN}http://localhost:3000${NC}\n"

# Wait indefinitely to keep the script alive
wait
