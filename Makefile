# Makefile for managing Quarto website

# Variables
QUARTO = quarto
OUTPUT_DIR = _site
PROJECT_DIR = .

# Default target: build the Quarto website
all: build

# Build the Quarto website
build:
	@echo "Building the website..."
	$(QUARTO) render $(PROJECT_DIR) --to html

# Preview the website (this starts a local server)
preview:
	@echo "Starting preview..."
	$(QUARTO) preview $(PROJECT_DIR)

# Clean the output directory
clean:
	@echo "Cleaning the output directory..."
	rm -rf $(OUTPUT_DIR)

# Publish the site (you can customize this for your host, e.g., GitHub Pages)
publish: build
	@echo "Publishing the website..."
	# Example: Publish to GitHub Pages (replace with your deploy command)
	# git add .
	# git commit -m "Update website"
	# git push origin main

# Watch for changes and automatically rebuild
watch:
	@echo "Watching for changes..."
	$(QUARTO) render $(PROJECT_DIR) --watch

# Help command to display options
help:
	@echo "Makefile options:"
	@echo "  make build    - Build the Quarto website"
	@echo "  make preview  - Preview the website locally"
	@echo "  make clean    - Clean the output directory"
	@echo "  make publish  - Build and publish the website"
	@echo "  make watch    - Watch for changes and rebuild"

