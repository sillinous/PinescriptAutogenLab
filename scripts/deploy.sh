#!/bin/bash

# deploy.sh - Deployment automation script for PinescriptAutogenLab

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PineScript AutoGen Lab Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi

    print_info "Prerequisites check passed"
}

# Validate environment file
validate_env() {
    print_info "Validating environment configuration..."

    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_warn ".env file not found"

        if [ -f "$PROJECT_ROOT/.env.docker" ]; then
            print_info "Copying .env.docker to .env"
            cp "$PROJECT_ROOT/.env.docker" "$PROJECT_ROOT/.env"
            print_warn "Please update .env with your actual configuration"
            exit 1
        else
            print_error "No environment configuration found"
            exit 1
        fi
    fi

    # Check for critical environment variables
    source "$PROJECT_ROOT/.env"

    if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "your_jwt_secret_key_here_minimum_32_characters" ]; then
        print_error "JWT_SECRET not configured in .env"
        exit 1
    fi

    if [ -z "$WEBHOOK_SECRET" ]; then
        print_error "WEBHOOK_SECRET not configured in .env"
        exit 1
    fi

    if [ -z "$ENCRYPTION_KEY" ]; then
        print_error "ENCRYPTION_KEY not configured in .env"
        exit 1
    fi

    print_info "Environment validation passed"
}

# Create necessary directories
setup_directories() {
    print_info "Setting up directories..."

    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/backups"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/nginx/ssl"

    print_info "Directories created"
}

# Generate self-signed SSL certificates (for development)
generate_ssl_certs() {
    if [ "$ENVIRONMENT" = "development" ]; then
        if [ ! -f "$PROJECT_ROOT/nginx/ssl/cert.pem" ]; then
            print_info "Generating self-signed SSL certificates..."

            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout "$PROJECT_ROOT/nginx/ssl/key.pem" \
                -out "$PROJECT_ROOT/nginx/ssl/cert.pem" \
                -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

            print_info "SSL certificates generated"
        fi
    fi
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."

    cd "$PROJECT_ROOT"

    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose --profile production build --no-cache
    else
        docker-compose build
    fi

    print_info "Docker images built successfully"
}

# Run database migrations
run_migrations() {
    print_info "Running database migrations..."

    # Initialize database if needed
    # This would run alembic migrations in a real setup
    # For now, the app auto-initializes the database

    print_info "Database ready"
}

# Start services
start_services() {
    print_info "Starting services..."

    cd "$PROJECT_ROOT"

    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose --profile production up -d
    else
        docker-compose up -d
    fi

    print_info "Services started"
}

# Wait for services to be healthy
wait_for_health() {
    print_info "Waiting for services to be healthy..."

    max_attempts=30
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health/ready &> /dev/null; then
            print_info "Backend is healthy"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    print_error "Services failed to become healthy"
    print_info "Checking logs..."
    docker-compose logs --tail=50
    exit 1
}

# Create initial admin user
create_admin() {
    print_info "Creating admin user..."

    # This would typically use a management command
    # For now, provide instructions
    print_warn "Please create admin user manually via API or database"
}

# Run tests
run_tests() {
    if [ "$ENVIRONMENT" = "development" ]; then
        print_info "Running tests..."

        docker-compose exec -T backend pytest -v || print_warn "Some tests failed"
    fi
}

# Display deployment info
display_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Services:"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Frontend: http://localhost:3000"
    echo ""
    echo "Useful commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart services: docker-compose restart"
    echo "  - Run tests: docker-compose exec backend pytest"
    echo ""

    if [ "$ENVIRONMENT" = "production" ]; then
        echo -e "${YELLOW}Production deployment notes:${NC}"
        echo "  - Configure proper SSL certificates in nginx/ssl/"
        echo "  - Update CORS_ORIGINS in .env"
        echo "  - Set up monitoring and alerting"
        echo "  - Configure automated backups"
        echo ""
    fi
}

# Rollback function
rollback() {
    print_warn "Rolling back deployment..."
    docker-compose down
    print_info "Rollback complete"
}

# Main deployment flow
main() {
    trap rollback ERR

    check_prerequisites
    validate_env
    setup_directories
    generate_ssl_certs
    build_images
    run_migrations
    start_services
    wait_for_health

    if [ "$ENVIRONMENT" = "development" ]; then
        run_tests
    fi

    display_info
}

# Run main function
main
