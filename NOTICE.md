Required Notice: Copyright Saul Dobney (https://github.com/dobneyresearch/PCNchip_with_leakyjug_learning)

# Licensing

> **Not legal advice.** These files were assembled to express a stated intent. Have them reviewed by
> a qualified adviser before you rely on them. The licensor name below is a placeholder pending
> confirmation of the correct legal entity.

This repository is **source-available, not open source**. The PolyForm licences are not
OSI-approved, and they deliberately restrict commercial use.

## Two licences, offered in parallel

| you are | your licence | what you may do |
|---|---|---|
| an academic, a public research body, a charity, a government institution, or an individual working on a noncommercial project | **[PolyForm Noncommercial 1.0.0](LICENSE.md)** | use, modify, and redistribute the software for any noncommercial purpose |
| a commercial company | **[PolyForm Free Trial 1.0.0](LICENSE-COMMERCIAL-EVALUATION.md)** | **evaluate only** — assess whether the software suits a particular application, for under 32 consecutive days. **No redistribution.** |
| a commercial company wanting anything beyond evaluation | **contact the licensor** | negotiated terms |

Both licence texts are the **unmodified**, canonical PolyForm texts. Neither has been edited; the
only addition is the `Required Notice:` line at the top of each, which the licences themselves
provide for. Offering them in parallel is permitted: each licence's *No Other Rights* section
expressly preserves the licensor's freedom to grant licences to anyone else.

## What this is intended to achieve

- **Academic and noncommercial users have full access**, including the right to modify and to
  redistribute their modifications.
- **Commercial companies may evaluate the system** and nothing more. Evaluation does not carry a
  right to distribute, to ship a product, or to use the design in production.
- **No downstream commercial use is possible without permission.** The Noncommercial licence permits
  redistribution only for noncommercial purposes, so a recipient cannot acquire commercial rights
  from an intermediary — those rights are retained by the licensor and are available only by
  agreement.

If the 32-day evaluation window in the Free Trial licence is too short for your assessment, ask; an
extension is a matter of correspondence rather than a different licence.

## What these licences do *not* cover

**The paper.** `paper/main_stage2_v4.pdf` and its LaTeX sources are a scholarly work, not software,
and a software licence is the wrong instrument for them. They are **© Saul Dobney, all rights
reserved**, except that quotation and citation under normal academic convention and applicable fair
use or fair dealing are unaffected. If the paper is posted to a preprint server it will carry
whatever licence that server requires, and that grant governs the copy hosted there.

**The Sky130A PDK.** The SPICE netlists in `Sky130A_16x16_4cell_jug/circuit/` are written against the
SkyWater Sky130A open PDK, which is licensed separately under Apache 2.0 by its own authors. The PDK
is not included in this repository and is not covered by these licences. You will need to obtain it
yourself to run the netlists.

**EMNIST.** The dataset is not included and is distributed by its own authors under its own terms
(Cohen et al., 2017).

**Third-party tooling.** `iverilog`, `ngspice`, NumPy, PyTorch, pandoc and TeX are all separately
licensed by their respective projects.

## Patents

Both licences grant a patent licence limited to the permitted purpose, and both terminate that
patent licence if you assert a patent claim against the software. No patent rights beyond the
permitted purpose are granted, and none are implied.

## Contact

Commercial licensing, evaluation extensions, and anything else: `saul.dobney@dobney.com`
