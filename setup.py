"""Setup script for PCIe Debug Agent"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path("README.md").read_text(encoding="utf-8")

# Read requirements
requirements = Path("requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

# Read dev requirements
dev_requirements = Path("requirements-dev.txt").read_text().splitlines()
dev_requirements = [r.strip() for r in dev_requirements if r.strip() and not r.startswith("#")]

setup(
    name="pcie-debug-agent",
    version="1.0.0",
    author="PCIe Debug Team",
    author_email="pcie-debug@example.com",
    description="AI-powered PCIe log analysis tool using RAG technology",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pcie_debug_agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "cli": ["templates/*.jinja2"],
    },
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "pcie-debug=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debuggers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    keywords="pcie debug log analysis rag ai llm",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pcie_debug_agent/issues",
        "Source": "https://github.com/yourusername/pcie_debug_agent",
        "Documentation": "https://pcie-debug-agent.readthedocs.io",
    },
)