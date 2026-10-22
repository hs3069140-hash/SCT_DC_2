#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  SkillCraft Technology - Task 02: Titanic EDA Runner"
echo "========================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3."
    exit 1
fi

echo "[1/2] Installing requirements..."
pip install -r requirements.txt --quiet

echo "[2/2] Running Titanic EDA..."
python3 titanic_eda.py

echo ""
echo "========================================================"
echo "  Done! Check the 'charts/' folder for all plots."
echo "========================================================"
