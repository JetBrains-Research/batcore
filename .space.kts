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
        env["URL"] = "https://packages.jetbrains.team/pypi/p/rrr/rrrpythonindex/simple"

        shellScript {
            content = """
                unzip beam.zip
                python run.py
            """
        }
    }
}

job("hyperparameters") {
    startOn {
        gitPush { enabled = false }
    }
    container(image = "registry.jetbrains.team/p/rrr/rrrpython/myimage:latest") {
        env["URL"] = "https://packages.jetbrains.team/pypi/p/rrr/rrrpythonindex/simple"

        shellScript {
            content = """
                unzip beam.zip
                echo running hyper.py
                python hyper.py
            """
        }
    }
}
