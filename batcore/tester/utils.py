def tester_logging(f):
    def wrapper(self,
                recommender,
                data_iterator,
                verbose=False,
                log_file_path=None,
                log_stdout=False,
                *args, **kwargs):
        self.setup_logger(verbose, log_file_path, log_stdout)
        try:
            return f(self,
                     recommender,
                     data_iterator,
                     verbose=False,
                     log_file_path=None,
                     log_stdout=False,
                     *args, **kwargs)
        except Exception as e:
            self.exception(f'got an exception {e}')
            raise

    return wrapper
