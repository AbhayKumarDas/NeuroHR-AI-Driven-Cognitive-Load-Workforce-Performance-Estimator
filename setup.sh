#!/bin/bash

# Create Streamlit config directory
mkdir -p ~/.streamlit/

# Create Streamlit config file
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor = '#ffb000'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F0F2F6'\n\
textColor = '#262730'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml
