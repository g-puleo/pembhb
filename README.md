### Installation


2. activate your favorite python environment with pytorch installed 

3. install the bbhx library by 
    ```
    pip install bbhx-cuda12x
    ```
    or for cpu only
    ```
    pip install bbhx
    ```

4.  clone this repository and install the package: 

    ```
    git clone https://github.com/g-puleo/pembhb.git
    cd pembhb
    pip install -e . 
    ```

### Usage 

See the `scripts/example.py` for an example of how to use the simulator and pack its output into a torch dataset.