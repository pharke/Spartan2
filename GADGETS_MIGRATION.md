# Gadgets 模块迁移指南

## 已完成的工作

1. ✅ 创建了 `constants` 模块，定义了 `BN_LIMB_WIDTH` 和 `BN_N_LIMBS`
2. ✅ 修复了所有 `crate::frontend` 导入，改为使用 `bellpepper_core`
3. ✅ 在 `lib.rs` 中导出了 `gadgets` 模块为公共模块
4. ✅ 修复了 `gadgets/mod.rs` 中的可见性，将 `pub(crate)` 改为 `pub`
5. ✅ 移除了 `ROCircuitTrait` 的引用（注释掉了 `absorb_in_ro` 函数）
6. ✅ 创建了 `OptionExt` helper trait 来支持 `.get()` 方法

## 还需要完成的工作

由于 gadgets 模块是从 Nova 库复制过来的，其中大量使用了 `.get_value().get()?` 模式。在 bellpepper_core 中，`get_value()` 返回 `Option<T>`，所以需要将：

```rust
// 旧代码 (Nova 风格)
let val = *some_num.get_value().get()?;

// 新代码 (bellpepper_core 风格)
let val = some_num.get_value().ok_or(SynthesisError::AssignmentMissing)?;
```

### 需要修复的文件

1. **src/gadgets/ecc.rs** - 约 50+ 处需要修复
2. **src/gadgets/utils.rs** - 约 10+ 处需要修复

### 修复模式

对于 `AllocatedNum`:
```rust
// 旧: *num.get_value().get()?
// 新: num.get_value().ok_or(SynthesisError::AssignmentMissing)?
```

对于 `AllocatedBit`:
```rust
// 旧: *bit.get_value().get()?
// 新: bit.get_value().ok_or(SynthesisError::AssignmentMissing)?
```

对于 `Boolean`:
```rust
// 旧: *bool.get_value().get()?
// 新: bool.get_value().ok_or(SynthesisError::AssignmentMissing)?
```

## 使用方法

一旦编译错误修复完成，你可以这样使用 ECC gadgets:

```rust
use spartan2::gadgets::ecc::AllocatedPoint;
use spartan2::traits::Engine;

// 在你的 circuit 中
let point = AllocatedPoint::<E>::default(cs.namespace(|| "point"))?;
point.check_on_curve(cs.namespace(|| "check"))?;
```

## 注意事项

- `absorb_in_ro` 函数已被注释掉，因为它依赖于 Nova 的 `RO2Circuit` trait
- 如果需要类似功能，可以使用 spartan2 的 transcript 系统重新实现
- 所有 gadgets 都基于 bellpepper 约束系统，与 spartan2 完全兼容

