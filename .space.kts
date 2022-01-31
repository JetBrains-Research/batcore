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

        push("https://jetbrains.team/p/rrr/packages/container/rrrpython") {
            tags("0.0.1")
        }
    }
}