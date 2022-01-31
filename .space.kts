job("Prepare Docker image") {
    // do not run on git push
    startOn {
        gitPush { enabled = false }
    }

    docker {
        build {
            file = "./Dockerfile"
            labels["vendor"] = "jbr"
        }

        push("registry.jetbrains.team/p/rrr/rrrpython/myimage") {
            tags("latest")
        }
    }
}


job("Run tests") {
    startOn {
        gitPush { enabled = false }
    }
    container(image = "registry.jetbrains.team/p/rrr/rrrpython/myimage:latest") {
        // specify URL of the package index using env var
        env["URL"] = "https://packages.jetbrains.team/pypi/p/rrr/rrrpythonindex/simple"

        // We suppose that your project has default build configuration -
        // the built package is saved to the ./dist directory
        shellScript {
            content = """
                unzip beam.zip
                mv beam github_csv
                python run.py
            """
        }
    }
}
