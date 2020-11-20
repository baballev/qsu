import os
import shutil

OSU_FOLDER_PATH = 'D:/Programmes/osu!/'
SWAP_FOLDER = 'E:/Programmation/Python/qsu!/backup_osu!/'
USERNAME = 'babal'
AI_NAME = 'OMBest'


def swap_to_AI(ai_name, username_beginning):
    try:
        print('Moving ' + OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg' + ' to: ' + SWAP_FOLDER + 'osu!.' + username_beginning + '.cfg')
        shutil.move(OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg', SWAP_FOLDER + 'osu!.' + username_beginning + '.cfg')
        print('Copying ' + os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'osu!' + ai_name + '.cfg' + ' to: ' + OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg')
        shutil.copy(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'osu!.' + ai_name + '.cfg', OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg')
        print('Swap to AI config: Success.')
    except Exception as e:
        print(e)
        print('Swap to AI config: Failure.')

def restore_player(username_beginning):
    if not os.path.exists(SWAP_FOLDER + 'osu!.' + username_beginning + '.cfg'):
        print('Cannot find the user config file: ' + SWAP_FOLDER + 'osu!.' + username_beginning + '.cfg')
        print('Restore player config: Failure')
        return

    try:
        print('Deleting ' + OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg')
        os.remove(OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg')
        print('Moving ' + SWAP_FOLDER + 'osu!.' + username_beginning + '.cfg' + ' to: ' + OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg')
        shutil.move(SWAP_FOLDER + 'osu!.' + username_beginning + '.cfg', OSU_FOLDER_PATH + 'osu!.' + username_beginning + '.cfg')
    except Exception as e:
        print(e)
        print('Restore player config: Failure')


if __name__ == '__main__':
    username = 'babal'
    AI_name = 'OMBest'
    #swap_to_AI(AI_name, username)
    #restore_player(username, AI_name)
