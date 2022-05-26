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

        push("registry.jetbrains.team/p/rrr/rrrpython/py37") {
            tags("latest")
        }
    }
}


job("Run zookeeper") {
    startOn {
        gitPush { enabled = false }
    }
    container(image = "registry.jetbrains.team/p/rrr/rrr-python37/myimage:latest") {
        env["URL"] = "https://packages.jetbrains.team/pypi/p/rrr/rrrpythonindex/simple"

        shellScript {
            content = """
                unzip zookeeper.zip
                python run.py
            """
        }
    }
}
