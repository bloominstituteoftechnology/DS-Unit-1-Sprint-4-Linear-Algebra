from os import path
import wget

def DownloadFile(url, name=None, target_dir=None):
    '''
    Check if file exists and download to working directory if it does not. Returns str of filename given to file.
        Arguments: url = str_of_fully_qualified_url, name=str_of_name_you_want
    TODO:
    2019-07-16 Check if suffix in name
    '''

    # Strip url for ending filename
    split_url = url_split = url.split('/')
    if len(split_url) > 2:
        filename = url_split[len(url_split)-1]
    else:
        filename = url_split[1]

    # Check if target_dir parameter given.  If so, append to new filename.
    if not target_dir:
        target_dir = ''
    else:
        if not path.isdir(target_dir):
            print('directory not found')
            return
    suffix = '.'.join(filename.split('.')[1:])
    # Check if name given
    if name:
        outpath = target_dir + name + '.' + suffix
    else:
        outpath = filename

    if path.exists(outpath):
        print('File already exists')
    else:
        try:
            filename = wget.download(url, out=outpath)
            print(filename, 'successfully downloaded to', outpath)
        except:
            print('File could not be downloaded.  Check URL & connection.')
    return filename


def UnzipFile(path, target_dir):
    '''
    Unpack compressed file to target directory.
        Arguments: path = str_of_path *relative or absolute, target_dir = str_of_path_to_dump
    '''
    from shutil import unpack_archive
    try:
        unpack_archive(path, target_dir)
    except:
        print('error unzipping')
    
