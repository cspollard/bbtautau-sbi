module MixtureModel where

open import Level using (Level; _⊔_) renaming (suc to lsuc)

-- record Generator (r i x : Level) : Set (lsuc (r ⊔ i ⊔ x)) where
--   field
--     R : Set r
--     I : Set i
--     X : Set x
--     gen : R → I → X

-- _⊞_ : ∀ {r r' i i' x x'} → 

open import Algebra using (Monoid)
module _ {w ℓ a : Level} (WM : Monoid w ℓ) (A : Set a) where
  open Monoid WM renaming (Carrier to W)
    renaming (_≈_ to _≈ʷ_; setoid to w-setoid)

  open import Data.List using (List; map; _++_)
  open import Data.Product using (_×_; _,_)
  open import Function using (id; const)
  open import Relation.Binary.PropositionalEquality
    using (_≡_) renaming (setoid to ≡-setoid)

  open import Data.Product.Relation.Binary.Pointwise.NonDependent
    renaming (×-setoid to bin-setoid)

  open import Relation.Binary.Bundles using (Setoid)

  ×-setoid : Setoid _ _
  ×-setoid = bin-setoid w-setoid (≡-setoid A)

  open import Data.List.Relation.Binary.Equality.Setoid ×-setoid
    using (_≋_; map⁺; ≋-setoid)

  Samples : Set (w ⊔ a)
  Samples = List (W × A)

  Sample : Set _
  Sample = Samples → List A

  reweight : (A → W) → Samples → Samples
  reweight rw = map λ where (w , a) → (rw a ∙ w) , a

  reweight-id : ∀ xs → reweight (const ε) xs ≋ xs
  reweight-id xs = {! map⁺ l-refl ?  !}
    where
      open Setoid ≋-setoid using () renaming (refl to l-refl)

  -- This is more general... doesn't need weighted samples for that.
  purturb : (A → A) → Samples → Samples
  purturb p = map λ where (w , a) → w , p a

  purturb-id : ∀ xs → purturb id xs ≡ xs
  purturb-id = map-id
    where open import Data.List.Properties

  Generator : ∀ {t r} (θ : Set t) (R : Set r) → Set (w ⊔ a ⊔ t ⊔ r)
  Generator θ r = θ → r → Samples

  mix : (S T : Samples) → Samples
  mix = _++_
