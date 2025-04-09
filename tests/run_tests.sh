#!/bin/bash
# PredictDb-Snakemake Test Runner

# Create required directories
mkdir -p tests/data tests/results tests/logs

# Output formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting PredictDb-Snakemake test suite${NC}"
echo "------------------------------------------------"

# Run unit tests
echo -e "${BLUE}Running unit tests...${NC}"
unit_tests_passed=true

for test_file in tests/unit/test_*.py; do
    if [ -f "$test_file" ]; then
        echo "Running $test_file"
        python "$test_file"
        if [ $? -ne 0 ]; then
            echo -e "${RED}UNIT TEST FAILED: $test_file${NC}"
            unit_tests_passed=false
        else
            echo -e "${GREEN}PASSED: $test_file${NC}"
        fi
    fi
done

# Skip integration tests if no test data is available
if [ -d "tests/data" ] && [ "$(ls -A tests/data)" ]; then
    echo -e "\n${BLUE}Running integration tests...${NC}"
    integration_tests_passed=true
    
    for test_file in tests/integration/*.py; do
        if [ -f "$test_file" ]; then
            echo "Running $test_file"
            python "$test_file"
            if [ $? -ne 0 ]; then
                echo -e "${RED}INTEGRATION TEST FAILED: $test_file${NC}"
                integration_tests_passed=false
            else
                echo -e "${GREEN}PASSED: $test_file${NC}"
            fi
        fi
    done
else
    echo -e "\n${BLUE}Skipping integration tests: no test data available in tests/data directory${NC}"
    integration_tests_passed=true
fi

# Print summary
echo -e "\n${BLUE}Test Summary:${NC}"
echo "------------------------------------------------"

if [ "$unit_tests_passed" = true ]; then
    echo -e "Unit tests: ${GREEN}PASSED${NC}"
else
    echo -e "Unit tests: ${RED}FAILED${NC}"
fi

if [ "$integration_tests_passed" = true ]; then
    echo -e "Integration tests: ${GREEN}PASSED${NC}"
else
    echo -e "Integration tests: ${RED}FAILED${NC}"
fi

# Exit with appropriate status code
if [ "$unit_tests_passed" = true ] && [ "$integration_tests_passed" = true ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
fi 