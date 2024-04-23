import sys
from pathlib import Path

file_path = f'{Path.home()}/Library/LaunchAgents/gitlab-runner.plist'

file_object = open(file_path, 'r')
content = file_object.readlines()
file_object.close()

host = str(sys.argv[1])
port = str(sys.argv[2])


for i, line in enumerate(content):
    if line.find('</array>') > 0:
        content.insert(i + 1, '<key>EnvironmentVariables</key>\n')
        content.insert(i + 2, '<dict>\n')
        content.insert(i + 3, '    <key>HTTP_PROXY</key>\n')
        content.insert(i + 4, f'    <string>{host}:{port}</string>\n')
        content.insert(i + 5, '    <key>HTTPS_PROXY</key>\n')
        content.insert(i + 6, f'    <string>{host}:{port}</string>\n')
        content.insert(i + 7, '</dict>\n')
        break

file_object = open(file_path, 'w')
contents = ''.join(content)
file_object.write(contents)
file_object.close()
