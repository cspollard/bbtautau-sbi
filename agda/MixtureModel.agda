module MixtureModel where

open import Level using (Level; _⊔_) renaming (suc to lsuc)

-- record Generator (r i x : Level) : Set (lsuc (r ⊔ i ⊔ x)) where
--   field
--     R : Set r
--     I : Set i
--     X : Set x
--     gen : R → I → X

-- _⊞_ : ∀ {r r' i i' x x'} → 

open import Algebra using (CommutativeRing)
module _ {w ℓ a : Level} (WR : CommutativeRing w ℓ) (A : Set a) where
  open CommutativeRing WR renaming (Carrier to W)
  open import Data.List using (List; map)
  open import Data.Product using (_×_; _,_)

  Samples : Set (w ⊔ a)
  Samples = List (W × A)

  reweight : (W × A → W) → Samples → Samples
  reweight rw = map λ where wa@(_ , a) → rw wa , a

  Generator : ∀ {t r} (θ : Set t) (R : Set r) → Set (w ⊔ a ⊔ t ⊔ r)
  Generator θ r = θ → r → Samples