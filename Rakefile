VERSION = File.open('VERSION').gets.strip
PYTHON_ENVS = [:env26, :env27, :env32, :env33]
PYTHON_EXECS = {:env26 => "python2.6", :env27 => "python2.7", :env32 => "python3.2", :env33 => "python3.3"}

def colorize(text, color)
  color_codes = {
    :black    => 30,
    :red      => 31,
    :green    => 32,
    :yellow   => 33,
    :blue     => 34,
    :magenta  => 35,
    :cyan     => 36,
    :white    => 37
  }
  code = color_codes[color]
  if code == nil
    text
  else
    "\033[#{code}m#{text}\033[0m"
  end
end

def virtual_env(command, env="env33")
  sh "source #{env}/bin/activate ; #{command}"
end

def create_virtual_env(dir, python)
  sh "virtualenv #{dir} -p #{python}"
end

task :clean => [] do
  sh "rm -rf ~*"
  sh "rm -rf *.pyc *.pyo"
  sh "rm -rf data/"
  sh "rm -rf *.egg-info"
  sh "rm -rf dist/"
end

task :install => [] do
  sh "python --version"
  sh "ruby --version"
  sh "easy_install pip"
end

task :dev_env => [] do
  PYTHON_ENVS.each { |env|
    puts colorize("Environment #{env}", :blue)
    create_virtual_env(env, PYTHON_EXECS[env])
  }
end

task :dependencies => [:dev_env] do
  PYTHON_ENVS.each { |env|
    puts colorize("Environment #{env}", :blue)
    virtual_env("pip install -r requirements.txt", "#{env}")
    virtual_env("pip install -r requirements-test.txt", "#{env}")
  }
end

task :tests => [] do
  PYTHON_ENVS.each { |env|
    puts colorize("Environment #{env}", :blue)
    virtual_env("nosetests", env)
  }
end

task :tag => [:tests] do
  sh "git tag #{VERSION}"
  sh "git push origin #{VERSION}"
end

task :reset_tag => [] do
  sh "git tag -d #{VERSION}"
  sh "git push origin :refs/tags/#{VERSION}"
end

task :publish => [:tests, :tag] do
  # http://guide.python-distribute.org/quickstart.html
  # python setup.py sdist
  # python setup.py register
  # python setup.py sdist upload
  # Manual upload to PypI
  # http://pypi.python.org/pypi/THE-PROJECT
  # Go to 'edit' link
  # Update version and save
  # Go to 'files' link and upload the file
  virtual_env("python setup.py sdist upload")
end

task :default => [:tests]

