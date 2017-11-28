import os
import sys
import getopt

def main(argv=None):
    isDemo = None
    # Handle arguments and options
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:],'d')
    except getopt.GetoptError as err:
        print(err.msg)
        return 2
    # Set global isDemo variable
    for o in opts:
        if o[0] == '-d':
            isDemo = True
        else:
            isDemo = False
    
    # Create demo conf file if necessary
    if isDemo:
        with open("DemoS3.conf","w") as f:
             lines = ["[DEFAULT]\n",
                      "AccessKeyID = {}\n".format(args[0]),
                      "AccessKeySecret = {}".format(args[1])]
             f.writelines(lines)
        with open("isDemo.conf","w") as f:
            lines = ["[DEFAULT]\n",
                      "isDemo = {}".format("true")]
            f.writelines(lines)
    else:
        with open("S3.conf","w") as f:
             lines = ["[DEFAULT]\n",
                      "AccessKeyID = {}\n".format(args[0]),
                      "AccessKeySecret = {}".format(args[1])]
             f.writelines(lines)
        with open("isDemo.conf","w") as f:
            lines = ["[DEFAULT]\n",
                      "isDemo = {}".format("false")]
            f.writelines(lines)
            
if __name__=='__main__':
    sys.exit(main())