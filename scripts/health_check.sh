#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════╗
# ║  Factory AI Brain — Health Check Script                      ║
# ║  Verifies all services are healthy and reachable             ║
# ╚══════════════════════════════════════════════════════════════╝

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
TOTAL=0

check_http() {
    local name="$1"
    local url="$2"
    local expected="${3:-healthy}"
    TOTAL=$((TOTAL + 1))

    printf "  %-30s " "${name}"
    if response=$(curl -sf --max-time 5 "$url" 2>/dev/null); then
        if echo "$response" | grep -q "$expected"; then
            printf "${GREEN}✓ HEALTHY${NC}\n"
            PASS=$((PASS + 1))
        else
            printf "${YELLOW}⚠ UNEXPECTED RESPONSE${NC}\n"
            FAIL=$((FAIL + 1))
        fi
    else
        printf "${RED}✗ UNREACHABLE${NC}\n"
        FAIL=$((FAIL + 1))
    fi
}

check_tcp() {
    local name="$1"
    local host="$2"
    local port="$3"
    TOTAL=$((TOTAL + 1))

    printf "  %-30s " "${name}"
    if nc -z -w 3 "$host" "$port" 2>/dev/null || (echo > /dev/tcp/"$host"/"$port") 2>/dev/null; then
        printf "${GREEN}✓ REACHABLE${NC}\n"
        PASS=$((PASS + 1))
    else
        printf "${RED}✗ UNREACHABLE${NC}\n"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Factory AI Brain — Service Health Check                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "${CYAN}── Backend Services ──${NC}"
check_http "API Gateway"            "http://localhost:8000/health"
check_http "RAG Service"            "http://localhost:8001/health"
check_http "Agent Service"          "http://localhost:8002/health"
check_http "Telemetry Service"      "http://localhost:8003/health"
check_http "Knowledge Graph Service" "http://localhost:8004/health"
check_http "Ingestion Service"      "http://localhost:8005/health"
echo ""

echo "${CYAN}── Frontend ──${NC}"
check_http "Next.js Frontend"       "http://localhost:3000" "html"
echo ""

echo "${CYAN}── Databases ──${NC}"
check_tcp  "PostgreSQL"             "localhost" 5432
check_tcp  "Redis"                  "localhost" 6379
check_tcp  "Neo4j (Bolt)"          "localhost" 7687
check_http "Neo4j (HTTP)"          "http://localhost:7474" ""
check_http "InfluxDB"              "http://localhost:8086/health" "pass"
check_http "Qdrant"                "http://localhost:6333/healthz" ""
echo ""

echo "${CYAN}── Messaging ──${NC}"
check_tcp  "Zookeeper"             "localhost" 2181
check_tcp  "Kafka"                 "localhost" 9092
echo ""

echo "${CYAN}── ML Platform ──${NC}"
check_http "MLflow"                "http://localhost:5000/health" ""
echo ""

echo "${CYAN}── Monitoring ──${NC}"
check_http "Prometheus"            "http://localhost:9090/-/healthy" "OK"
check_http "Grafana"               "http://localhost:3001/api/health" "ok"
check_http "Jaeger"                "http://localhost:16686" ""
echo ""

echo "════════════════════════════════════════════════════════════"
printf "  Results: ${GREEN}%d passed${NC} / ${RED}%d failed${NC} / %d total\n" "$PASS" "$FAIL" "$TOTAL"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "${RED}⚠  Some services are unhealthy. Check docker logs for details.${NC}"
    exit 1
else
    echo "${GREEN}✅ All services are healthy!${NC}"
    exit 0
fi
