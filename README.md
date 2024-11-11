# Tutorial Apps

Central repository designed to host various tutorial applications, each maintained in its dedicated branch for organization and ease of access.

Apps List:

- [Basic Aggregator](https://github.com/OpenMined/tutorial-apps/tree/main/basic_aggregator): Collects and summarizes public data from multiple data sites.

## Install a tutorial app

Installing SyftBox apps is equivalent to copying an app's directory (or creating a link) in your client's sync directory, under `apps`.

You can do this in 2 ways:

- use the `syftbox` CLI
- use basic shell commands

### SyftBox CLI

1. clone this repository somewhere on your computer

```sh
git clone https://github.com/OpenMined/tutorial-apps.git
```

2. run `syftbox app install` with the app you want to install

```sh
syftbox app install tutorial-apps/basic_aggregator
```

the first argument should be the path to the app's directory

### Shell commands

1. clone this repository

```sh
git clone https://github.com/OpenMined/tutorial-apps.git
```

2. create a symlink inside your SyftBox sync folder, under `apps`

```sh
# change this to reflect the correct path based on your setup
APPS_DIR=~/SyftBox/apps

# note: using absolute path for the source
ln -s $(pwd)/tutorial-apps/basic_aggregator $APPS_DIR/apps
```
