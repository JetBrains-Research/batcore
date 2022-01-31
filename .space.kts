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