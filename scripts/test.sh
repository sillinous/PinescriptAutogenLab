#!/bin/bash

# test.sh - Test runner script for PinescriptAutogenLab

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running PineScript AutoGen Lab Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Test types
TEST_TYPE="${1:-all}"

# Function to run tests
run_tests() {
    local test_marker="$1"
    local test_name="$2"

    echo -e "${GREEN}Running $test_name...${NC}"

    if [ "$test_marker" = "all" ]; then
        pytest -v --cov=backend --cov-report=html --cov-report=term
    else
        pytest -v -m "$test_marker" --cov=backend --cov-report=html --cov-report=term
    fi

    echo ""
}

# Install test dependencies
echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -q pytest pytest-asyncio pytest-cov

# Run tests based on type
case $TEST_TYPE in
    "unit")
        run_tests "unit" "Unit Tests"
        ;;
    "integration")
        run_tests "integration" "Integration Tests"
        ;;
    "e2e")
        run_tests "e2e" "End-to-End Tests"
        ;;
    "security")
        run_tests "security" "Security Tests"
        ;;
    "websocket")
        run_tests "websocket" "WebSocket Tests"
        ;;
    "trading")
        run_tests "trading" "Trading Tests"
        ;;
    "fast")
        echo -e "${YELLOW}Running fast tests (excluding slow tests)...${NC}"
        pytest -v -m "not slow" --cov=backend --cov-report=term
        ;;
    "all"|*)
        run_tests "all" "All Tests"
        ;;
esac

# Display coverage report location
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Tests Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Coverage report: htmlcov/index.html"
echo ""

# Check coverage threshold
coverage_percent=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')

if [ $(echo "$coverage_percent >= 80" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Coverage: $coverage_percent% (meets 80% threshold)${NC}"
else
    echo -e "${YELLOW}! Coverage: $coverage_percent% (below 80% threshold)${NC}"
fi
