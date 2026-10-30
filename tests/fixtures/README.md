`crid_v1_vectors.json` is copied byte-for-byte from
[layervai/qurl-conformance](https://github.com/layervai/qurl-conformance/blob/308896f/vectors/crid_v1_vectors.json)
at commit `308896f`. It pins the same public CRID contract consumed by qurl-go.

The existing conformance schema-1 CI job installs qurl-conformance 0.11.0,
which does not contain the CRID accessor/artifact. Keeping this fixture local
lets that compatibility job keep exercising the older qv2 artifact while
always running the CRID gate and key-binding vectors. Do not regenerate or
edit the bytes here; copy an intentionally reviewed upstream revision.
