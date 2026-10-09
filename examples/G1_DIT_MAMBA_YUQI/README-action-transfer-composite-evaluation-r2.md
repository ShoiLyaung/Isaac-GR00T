# CPU fixture construction repair (r2)

This isolated branch starts at `c3f36d0f16821a9a868e28838d6648cede2c2ae6`. It preserves the r1 source and failed `g1-mamba-action-transfer-composite-evaluation-r1/materialize-initial-fixture-r1` attempt. That attempt made zero updates and failed while constructing the pretrained Qwen3-VL backbone because FlashAttention2 is unavailable on CPU; it did not produce a candidate.

The source path is `Gr00tN1d7.from_pretrained` → `Gr00tN1d7.__init__` → `Qwen3Backbone.__init__` → `Qwen3VLForConditionalGeneration.from_pretrained`. B0's on-disk `use_flash_attention=true` makes `Qwen3Backbone` pass `attn_implementation="flash_attention_2"` whenever the package is installed, independently of the fixture's intentionally hidden CUDA devices.

Only the private CPU fixture entrypoint now calls `strict_cpu_fixture_reload`. It uses the existing native per-load `use_flash_attention=False` config override. HF applies it to a fresh config for that load; Qwen3Backbone omits the forced FA2 keyword and the pinned Qwen3-VL selects its supported SDPA default. The returned backbone must report `sdpa`, or the fixture fails. This avoids a global class/factory monkeypatch and has no mutable scope requiring restoration. The B0 config file, following ordinary `strict_reload` calls, GPU default attention backend and action sampler remain unchanged.

The CLI remains:

```text
CUDA_VISIBLE_DEVICES='' .../python action_transfer_checkpoint.py fixture --plan PATH --plan-sha256 SHA --artifact-root ROOT --output OUT --wall-seconds N
```

No PLAN fields are added. The CPU entrypoint additionally requires the exact existing B0 semantic hash `2bff6e15f09adc0ee05c24d2445d347113d802043e41227e4eb6b6ce150a83ad`. After loading and converting to frozen CPU BF16, the unchanged fixture builder checks the full actual state dictionary against that hash before core/periphery assembly. Native strict-loading errors still fail; pretrained weights cannot be skipped, and hidden/uninitialized CUDA is required before and after construction. Loading metadata records `cpu_fixture_attention_backend="sdpa"` and the per-load override scope. No input is encoded and no optimizer/RNG training recovery occurs.

Focused CPU tests verify that the override does not modify the config file or subsequent normal-load defaults, including factory exceptions, missing weights and unexpected backend rejection. They also verify skip-weight/CUDA guards and load a real tiny pretrained Qwen3-VL through the actual Qwen3Backbone CPU path with bitwise-equal weights and no CUDA initialization. This is constructor compatibility evidence only. The complete B0 hash and approximately2.779GiB real fixture remain subject to the Supervisor's new explicit CPU attempt; the backend repair does not establish or alter any GPU execution mathematics.
