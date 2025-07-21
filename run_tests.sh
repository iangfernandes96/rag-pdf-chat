#!/bin/bash

# RAG PDF Chat - Test Runner Script

echo "🧠 RAG PDF Chat - Running Phase 2 & 3 Tests"
echo "=" * 50

# Test 1: Simple Phase 2 functionality
echo "🧩 Test 1: Core Components (models, chunking, config)"
echo "-" * 50
uv run python test_phase2_simple.py
test1_result=$?

echo -e "\n"

# Test 2: Full document ingestion with PDF
echo "📄 Test 2: Full Document Ingestion Pipeline"
echo "-" * 50
uv run python test_document_ingestion.py
test2_result=$?

echo -e "\n"

# Test 3: Phase 3 embedding and vector functionality
echo "🧮 Test 3: Embedding Generation and Vector Operations"
echo "-" * 50
uv run python test_phase3_embeddings.py
test3_result=$?

echo -e "\n"

# Summary
echo "📋 Test Summary"
echo "=" * 30

if [ $test1_result -eq 0 ]; then
    echo "✅ Core Components Test: PASSED"
else
    echo "❌ Core Components Test: FAILED"
fi

if [ $test2_result -eq 0 ]; then
    echo "✅ Document Ingestion Test: PASSED"
else
    echo "❌ Document Ingestion Test: FAILED"
fi

if [ $test3_result -eq 0 ]; then
    echo "✅ Embedding & Vector Test: PASSED"
else
    echo "❌ Embedding & Vector Test: FAILED"
fi

if [ $test1_result -eq 0 ] && [ $test2_result -eq 0 ] && [ $test3_result -eq 0 ]; then
    echo -e "\n🎉 All Phase 2 & 3 tests passed! Ready for Phase 4."
    exit 0
else
    echo -e "\n❌ Some tests failed. Please check the output above."
    exit 1
fi 