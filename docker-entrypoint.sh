#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 PCIe Debug Agent - Docker Container Starting...${NC}"

# Function to check if a service is available
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}⏳ Waiting for $service to be ready...${NC}"
    
    while ! nc -z $host $port 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}❌ $service is not available after $max_attempts attempts${NC}"
            return 1
        fi
        echo -n "."
        sleep 1
    done
    
    echo -e "\n${GREEN}✅ $service is ready!${NC}"
    return 0
}

# Check environment variables
check_env_vars() {
    local required_vars=()
    local missing_vars=()
    
    # Add your required environment variables here
    if [ "$APP_ENV" = "production" ]; then
        required_vars+=("OPENAI_API_KEY" "EMBEDDING_API_KEY")
    fi
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=($var)
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing required environment variables:${NC}"
        printf '%s\n' "${missing_vars[@]}"
        echo -e "${YELLOW}Please set these variables in your .env file${NC}"
        exit 1
    fi
}

# Initialize directories
init_directories() {
    echo -e "${YELLOW}📁 Initializing directories...${NC}"
    
    # Create necessary directories if they don't exist
    mkdir -p /app/data/vectorstore /app/logs /app/reports /app/configs
    
    # Set proper permissions
    chmod -R 755 /app/data /app/logs /app/reports /app/configs
    
    echo -e "${GREEN}✅ Directories initialized${NC}"
}

# Wait for dependent services
wait_for_dependencies() {
    # Wait for Redis if enabled
    if [ ! -z "$REDIS_HOST" ]; then
        wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
    fi
    
    # Wait for PostgreSQL if enabled
    if [ ! -z "$POSTGRES_HOST" ]; then
        wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}     PCIe Debug Agent - Enhanced UI${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "Environment: ${APP_ENV:-development}"
    echo -e "Log Level: ${LOG_LEVEL:-INFO}"
    echo -e "Port: ${STREAMLIT_SERVER_PORT:-8501}"
    echo -e "${GREEN}================================================${NC}\n"
    
    # Check environment variables
    check_env_vars
    
    # Initialize directories
    init_directories
    
    # Wait for dependencies
    wait_for_dependencies
    
    # Start the application
    echo -e "${GREEN}🎯 Starting Streamlit application...${NC}"
    echo -e "${YELLOW}📍 Access the application at: http://localhost:${STREAMLIT_SERVER_PORT:-8501}${NC}\n"
    
    # Execute the main application
    exec streamlit run src/ui/app_refactored.py \
        --server.port=${STREAMLIT_SERVER_PORT:-8501} \
        --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
        --server.headless=${STREAMLIT_SERVER_HEADLESS:-true} \
        --browser.gatherUsageStats=${STREAMLIT_BROWSER_GATHERUSAGESTATS:-false} \
        --theme.base=${STREAMLIT_THEME_BASE:-dark} \
        --theme.primaryColor=${STREAMLIT_THEME_PRIMARYCOLOR:-#1f77b4} \
        --server.enableCORS=${STREAMLIT_SERVER_ENABLE_CORS:-false} \
        --server.enableXsrfProtection=${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION:-true}
}

# Handle signals gracefully
trap 'echo -e "\n${YELLOW}⚠️  Shutting down gracefully...${NC}"; exit 0' SIGTERM SIGINT

# Run main function
main