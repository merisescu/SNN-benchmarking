#!/bin/bash
set -e

TEST=false
while getopts "t" opt; do
    case "$opt" in
        t)
            TEST=true
            ;;
        *)
            echo "Usage: $0 [-t]"
            exit 1
            ;;
    esac
done

if [ ! -d "/workspace/nest-src" ]; then
    git clone https://github.com/merisescu/nest-simulator /workspace/nest-src
    git -C /workspace/nest-src pull origin staging
    uv pip install --system -r /workspace/nest-src/requirements.txt
fi

if [ -f "/workspace/make_my_ext.log" ]; then
    rm make_my_ext.log
fi
cmake -S /workspace/nest-src -B /workspace/nest-build -DCMAKE_INSTALL_PREFIX="${NEST_INSTALL_DIR}" -Dwith-debug=ON | tee >(grep 'eprop_izhikevich' >> make_my_ext.log)
make -C /workspace/nest-build | tee >(grep 'eprop_izhikevich' >> make_my_ext.log)
make -C /workspace/nest-build install | tee >(grep 'eprop_izhikevich' >> make_my_ext.log)
if $TEST; then
make -C /workspace/nest-build installcheck | tee >(grep 'eprop_izhikevich' >> make_my_ext.log)
fi