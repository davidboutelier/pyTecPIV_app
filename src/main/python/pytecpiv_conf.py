def pytecpiv_get_pref():
    """
    This function checks the existence of a settings file in the working directory and returns the full paths to the
    sources and projects directory. If the settings files does not exist these paths are returned empty.
    :return: sources_path, projects_path - the full paths to the sources and the projects directory respectively
    """

    import os.path
    import json

    t = os.path.isfile('pytecpiv_settings.json')

    if t:
        with open('pytecpiv_settings.json') as f:
            pytecpiv_settings = json.load(f)

            sources = pytecpiv_settings['sources']
            sources_path = sources['sources_path']

            projects = pytecpiv_settings['projects']
            projects_path = projects['projects_path']

            file_exist = 'yes'

    else:
        file_exist = 'no'
        sources_path = ''
        projects_path = ''
    return file_exist, sources_path, projects_path

def pytecpiv_set_cores(fraction_cores):
    """

    :param fraction_cores:
    :return:
    """
    import multiprocessing

    #  get the number of cores available
    n_cores = multiprocessing.cpu_count()
    use_cores = int(fraction_cores * n_cores)

    if use_cores < 1:
        use_cores == 1

    return n_cores, use_cores









