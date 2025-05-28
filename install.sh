#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

printf "\033]0;Installer\007"
clear
rm -f *.bat  

# Function to log messages with timestamps
log_message() {
    local msg="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $msg"
}

# Function to find a suitable Python version
find_python() {
    for py in python3.10 python3 python; do
        if command -v "$py" > /dev/null 2>&1; then
            echo "$py"
            return
        fi
    done
    log_message "No compatible Python installation found. Please install Python 3.10."
    exit 1
}

# Function to create or activate a virtual environment
prepare_install() {
    if [ -d "venv" ]; then
        log_message "Virtual environment found. This has been already installed or this is a broken install."
        printf "Do you want to execute install.sh? (Y/N): " >&2
        read -r r
        r=$(echo "$r" | tr '[:upper:]' '[:lower:]')
        if [ "$r" = "y" ]; then
            chmod +x install.sh
            ./install.sh && exit 0
        else
            log_message "Continuing with the installation."
            rm -rf venv
            create_venv
        fi
    else
        create_venv
    fi
}

# Function to create the virtual environment and install dependencies
create_venv() {
    log_message "Creating virtual environment..."
    py=$(find_python)

    "$py" -m venv venv

    log_message "Activating virtual environment..."
    source venv/bin/activate

    # Install pip if necessary and upgrade
    log_message "Ensuring pip is installed..."
    python -m ensurepip --upgrade || {
        log_message "ensurepip failed, attempting manual pip installation..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
    }
    python -m pip install --upgrade pip

    log_message "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        python -m pip install -r requirements.txt
    else
        log_message "requirements.txt not found. Please ensure it exists."
        exit 1
    fi

    log_message "Installing dependencies..."
    python -m pip install -U diffusers transformers accelerate
    finish
}


prepare_install
