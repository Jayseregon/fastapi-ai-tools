# fastapi-ai-tools

[![Version](https://img.shields.io/badge/version-0.2.4.0228-blue)](https://github.com/jayseregon/programming/fastapi/fastapi-ai-tools)
[![Last Updated](https://img.shields.io/badge/last%20updated-2025.02.28-brightgreen)](https://github.com/jayseregon/programming/fastapi/fastapi-ai-tools)

**fastapi-ai-tools** is an internal FastAPI-based web API designed to integrate AI and RAG tools with Python for company use. This repository is private and intended exclusively for internal purposes. Its deployed to Azure under **RAG AI Toolbox**.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
  - [Directory Structure](#directory-structure)
  - [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Notes](#notes)
- [Contact](#contact)

## Overview
fastapi-ai-tools provides a robust API to support advanced AI workflows and Retrieval Augmented Generation (RAG) scenarios, built with performance and extensibility in mind.

## Features
- **AI Integration:** Seamlessly connects to various AI tools.
- **RAG Support:** Enhances retrieval augmented generation processes.
- **Asynchronous & Scalable:** Built on FastAPI to support high-performance asynchronous operations.
- **Modular Design:** Easily extensible for future enhancements.

## Installation
1. Clone the repository onto your internal server.
2. Create and activate a Python virtual environment (use virtualenv or conda).
3. Install dependencies:
   ```bash
   poertry install
   ```
4. Ensure required environment variables are set (see Configuration).

## Configuration
Configuration files reside under `/src/configs`. Update the `.env` file and other configuration settings as needed to match your local environment.

## Usage
To run the server:
```bash
fastapi dev src/main.py
```
Then visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

## Development

### Directory Structure
```plaintext
├── src/
│   ├── main.py         // FastAPI entry point
│   ├── configs/        // app env & log configs
│   ├── models/         // Pydantic models
│   ├── routes/         // app routers
│   ├── security/       // app security services
│   ├── services/       // core logic
│   └── tests/          // Unit and integration tests
├── pyproject.toml      // poetry config
├── Dockerfile          // production build
├── docker-compose.yml  // local redis instance
├── LICENSE
└── README.md
```

### Testing
Run tests using:
```bash
pytest
```
Ensure tests pass before committing changes.

## Contributing
Since this is a private project, please adhere to our company development guidelines:
- Make sure code changes are covered by tests.
- Update documentation as needed.
- Follow the internal code review processes before merging.

## License
This project is proprietary and for internal company use only. Distribution outside the company is strictly prohibited.

## Notes
- This repository is private and its contents are confidential.
- For inquiries or suggestions, please reach out to the internal development team.

## Contact

If you have any questions or need further clarification, feel free to reach out to the head developer of this project:

>  `Jeremie Bitsch`

Contact can be done via Teams or email. Please use the appropriate channel based on the nature of your query.
